import traceback
from global_cfgs import Global_Cfgs
from scenario import Scenario
from UIs.console_UI import Console_UI


def main():
    cfgs = Global_Cfgs()
    scenario = Scenario(scenario_name=cfgs.get('scenario'))

    start_scene = cfgs.get('start_scene')
    try:
        for scene in iter(scenario):
            if start_scene is None or scene.scene_name.strip().lower() == start_scene.strip().lower():
                start_scene = None
                scene.run_scene()
            else:
                Console_UI().inform_user(f'Skip \'{scene.scene_name}\' - waiting for \'{start_scene}\'')

    except RuntimeError as error:
        Console_UI().warn_user(error)
        Console_UI().inform_user("\n\n Traceback: \n")

        traceback.print_exc()
    except KeyboardInterrupt:
        Console_UI().inform_user(f'\nInterrupted by ctrl+c - stopped ad "{scene.scene_name}"')
    else:
        Console_UI().inform_user("Done with all scenarios!")

    Console_UI().inform_user('To view results, checkout the tensorboard:')
    Console_UI().inform_user(f'tensorboard --logdir /media/max/HD_1_3TB/log/{cfgs.sub_log_path}/tensorboard')


if __name__ == "__main__":
    main()
